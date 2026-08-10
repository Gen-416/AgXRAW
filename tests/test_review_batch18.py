# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 18 regression gates.

1. [P0] The GUI export filename call must not carry the seed (it crashed
   every export with TypeError).
2. The export child gets its seed RESOLVED BY THE PARENT — a spawned
   process has a fresh PREVIEW_STORE and would otherwise mint a new one.
3. Conservative scatter with a source taken from the FULL-RESOLUTION
   pre-bloom print: a sparse highlight sheds energy AND its neighbourhood
   gains it (the decimated-proxy source gave up energy to nobody).
5. inner budget 1 silences every nested pool, not just the outer one.
6. The GUI export releases its analysis buffers when no dashboard follows.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from tests.test_review_batch16 import _negative_stock, _plan


class ExportFilenameTests(unittest.TestCase):
    def test_suffix_call_does_not_pass_the_seed(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        call = src[src.find("suffix = export_suffix_parts("):]
        call = call[:call.find(")\n")]
        self.assertNotIn(
            "film_optics_seed", call,
            "export_suffix_parts has no such parameter — passing it raised "
            "TypeError on EVERY GUI export (review batch 18 P0)",
        )
        fingerprint = src[src.find("fingerprint = export_plan_fingerprint("):]
        fingerprint = fingerprint[:fingerprint.find(")\n")]
        self.assertIn(
            "film_optics_seed", fingerprint,
            "the seed must still separate two different renders' paths",
        )

    def test_suffix_signature_rejects_the_seed(self) -> None:
        from dngscan.gui.service import export_suffix_parts

        with self.assertRaises(TypeError):
            export_suffix_parts("clip", "srgb", "sdr", film_optics_seed=7)


class SpawnSeedTests(unittest.TestCase):
    def test_parent_resolves_the_seed_before_spawning(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export_isolated)
        resolve = src.find("filmOpticsSeed")
        spawn = src.find('mp.get_context("spawn")')
        self.assertGreater(resolve, 0, "the parent must resolve the seed")
        self.assertLess(
            resolve, spawn,
            "resolution must happen BEFORE the child is created — a spawned "
            "child has a fresh PREVIEW_STORE and would mint a new seed",
        )

    def test_resolved_seed_reaches_the_child_payload(self) -> None:
        from unittest import mock

        from dngscan.gui import service

        captured: dict = {}

        class _FakeProcess:
            def __init__(self, target=None, args=(), name=None):
                captured["params"] = args[0]

            def start(self):
                raise RuntimeError("stop here: the payload is what we check")

            def is_alive(self):
                return False

            def join(self, timeout=None):
                return None

        class _FakeCtx:
            def Queue(self, maxsize=0):
                return mock.Mock()

            Process = _FakeProcess

        params = {"input": "/tmp/x.dng", "outdir": "/tmp"}
        with mock.patch.object(service.mp, "get_context", lambda kind: _FakeCtx()):
            try:
                service.run_export_isolated(params)
            except Exception:
                pass
        self.assertIn("params", captured)
        seed = captured["params"].get("filmOpticsSeed")
        self.assertIsInstance(seed, int)
        self.assertGreater(seed, 0)


class ScatterSourceTests(unittest.TestCase):
    def test_sparse_highlight_sheds_and_the_neighbourhood_gains(self) -> None:
        """A decimated proxy source subtracted at full resolution stole light
        from pixels that never had any: a 1.0 highlight fell to 0.912 with
        nothing anywhere gained. Source and subtraction must be the SAME
        full-resolution quantity.

        Driven on the operator directly since P3: `film_bloom` now means the
        additive editorial capture bloom, which deliberately does NOT shed
        from the core, so routing this through the amount would assert the
        wrong operator's physics.
        """
        from dngscan.film_optics import (
            bloom_apply_rows,
            integral_from_field,
            scatter_source,
            scatter_spread,
        )
        from dngscan.film_optics_assets import (
            DEFAULT_PRINT_OPTICS,
            load_print_optics,
        )

        scatter = load_print_optics(DEFAULT_PRINT_OPTICS).print_scatter
        h, w = 64, 96
        img = np.full((h, w, 3), 0.01, dtype=np.float32)
        img[h // 2, w // 2] = 20.0
        spread_ii = integral_from_field(
            scatter_spread(scatter_source(img, scatter), scatter)
        ).astype(np.float32)
        out = bloom_apply_rows(
            img.reshape(-1, 3), spread_ii, 0, h, h, w, scatter, 1.0
        ).reshape(h, w, 3)
        luma = np.array([0.2627, 0.6780, 0.0593])
        self.assertLess(
            float(out[h // 2, w // 2] @ luma), float(img[h // 2, w // 2] @ luma),
            "the core must shed energy",
        )
        self.assertGreater(
            float(out[h // 2 + 3, w // 2] @ luma),
            float(img[h // 2 + 3, w // 2] @ luma),
            "the neighbourhood must RECEIVE what the core shed — a decimated "
            "proxy source gave it to nobody",
        )

    def test_bloom_applies_exactly_once(self) -> None:
        """Pass B renders the pre-bloom print with the same context; the
        map is None then, so bloom cannot be applied twice."""
        from dngscan.film_develop import prepare_film_spatial

        ctx = prepare_film_spatial(_plan(_negative_stock(), film_bloom=1.0), 32, 48)
        self.assertIsNotNone(ctx)
        self.assertIsNone(ctx.bloom_map, "the map must not exist before pass B")


class NestedBudgetTests(unittest.TestCase):
    def test_share_of_one_silences_every_pool(self) -> None:
        from unittest import mock

        from dngscan import agx, gated_drt, look
        from dngscan.cpu_budget import TOTAL, inner
        from dngscan.render import apply_tone_core
        from dngscan.tone import build_render_plan
        from tests.golden_support import build_daylight_wide_dr

        scene = build_daylight_wide_dr()
        submits: list[str] = []

        def spy(pool, name):
            real = pool.submit
            return lambda *a, **k: (submits.append(name), real(*a, **k))[1]

        rgb = np.random.default_rng(0).uniform(
            0.01, 0.9, (200_000, 3)
        ).astype(np.float32)
        plan = build_render_plan(scene.bundle, scene.analysis, "agx", "srgb")
        gated = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb", tone_core="gated"
        )
        lab = rgb.astype(np.float32)
        look_name = next(iter(look.LOOK_FIELDS))

        def exercise() -> None:
            apply_tone_core(rgb, plan.tone, plan.color, None, None)
            apply_tone_core(rgb, gated.tone, gated.color, None, None)
            look.apply_look_oklab(lab[:, 0], lab[:, 1], lab[:, 2], look_name, 1.0)

        with mock.patch.object(
            agx._FORMATION_POOL, "submit", spy(agx._FORMATION_POOL, "formation")
        ), mock.patch.object(
            look._LOOK_POOL, "submit", spy(look._LOOK_POOL, "look")
        ), mock.patch.object(
            gated_drt._GATED_POOL, "submit", spy(gated_drt._GATED_POOL, "gated")
        ):
            with inner(1):
                exercise()
            starved = list(submits)
            submits.clear()
            with inner(max(TOTAL, 2)):
                exercise()
            full = list(submits)
        self.assertEqual(
            starved, [],
            "a share of 1 must silence the formation, look and gated pools — "
            "the review measured 1 and 4 stray submits",
        )
        self.assertGreater(len(full), 0, "the pools must still work when budgeted")


class GuiStagedReleaseTests(unittest.TestCase):
    def test_gui_export_releases_before_encoding(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        release = src.find("release_analysis_buffers")
        encode = src.find("export_result = dg.export_jpeg")
        self.assertGreater(release, 0, "the GUI export must stage the release")
        self.assertLess(
            release, encode,
            "release must precede the JPEG/HDR encode (the dashboard holds "
            "its own export slot earlier, which is not the anchor here)",
        )
        # batch 19: the dashboard now runs BEFORE the export, so the
        # release is unconditional — a png=1 export used to encode with
        # xyz_render / y / ev_img still resident.
        dashboard = src.find("plot_dashboard")
        self.assertGreater(dashboard, 0)
        self.assertLess(
            dashboard, release,
            "the dashboard (the last consumer) must run before the release",
        )


if __name__ == "__main__":
    unittest.main()
